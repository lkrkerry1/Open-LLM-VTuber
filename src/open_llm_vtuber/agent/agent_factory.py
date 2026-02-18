from typing import Type, Literal
from loguru import logger
import importlib  # 新增

from .agents.agent_interface import AgentInterface
from .agents.basic_memory_agent import BasicMemoryAgent
from .stateless_llm_factory import (
    LLMFactory as StatelessLLMFactory,
)
from .agents.hume_ai import HumeAIAgent
from .agents.letta_agent import LettaAgent

from ..mcpp.tool_manager import ToolManager
from ..mcpp.tool_executor import ToolExecutor
from typing import Optional


class AgentFactory:
    @staticmethod
    def create_agent(
        conversation_agent_choice: str,
        agent_settings: dict,
        llm_configs: dict,
        system_prompt: str,
        live2d_model=None,
        tts_preprocessor_config=None,
        **kwargs,
    ) -> Type[AgentInterface]:
        """Create an agent based on the configuration.

        Args:
            conversation_agent_choice: The type of agent to create
            agent_settings: Settings for different types of agents
            llm_configs: Pool of LLM configurations
            system_prompt: The system prompt to use
            live2d_model: Live2D model instance for expression extraction
            tts_preprocessor_config: Configuration for TTS preprocessing
            **kwargs: Additional arguments
        """
        logger.info(
            f"Initializing agent: {conversation_agent_choice}"
        )

        if (
            conversation_agent_choice
            == "basic_memory_agent"
        ):
            # Get the LLM provider choice from agent settings
            basic_memory_settings: dict = (
                agent_settings.get("basic_memory_agent", {})
            )
            llm_provider: str = basic_memory_settings.get(
                "llm_provider"
            )

            if not llm_provider:
                raise ValueError(
                    "LLM provider not specified for basic memory agent"
                )

            # Get the LLM config for this provider
            llm_config: dict = llm_configs.get(llm_provider)
            # 🔧 修复：先检查 llm_config 是否为 None
            if llm_config is None:
                raise ValueError(
                    f"Configuration not found for LLM provider: {llm_provider}"
                )

            interrupt_method: Literal["system", "user"] = (
                llm_config.pop("interrupt_method", "user")
            )

            # 原来的 if not llm_config: 检查可以保留，但此时 llm_config 已经不为 None
            # 可选：如果 llm_config 在 pop 后变为空，是否需要处理？通常不需要，因为后面还会传给 create_llm
            # 但保留原有逻辑也无妨
            if not llm_config:
                logger.warning(
                    f"LLM config for {llm_provider} is empty after pop"
                )

            # Create the stateless LLM
            llm = StatelessLLMFactory.create_llm(
                llm_provider=llm_provider,
                system_prompt=system_prompt,
                **llm_config,
            )

            tool_prompts = kwargs.get(
                "system_config", {}
            ).get("tool_prompts", {})

            # Extract MCP components/data needed by BasicMemoryAgent from kwargs
            tool_manager: Optional[ToolManager] = (
                kwargs.get("tool_manager")
            )
            tool_executor: Optional[ToolExecutor] = (
                kwargs.get("tool_executor")
            )
            mcp_prompt_string: str = kwargs.get(
                "mcp_prompt_string", ""
            )

            # Create the agent with the LLM and live2d_model
            return BasicMemoryAgent(
                llm=llm,
                system=system_prompt,
                live2d_model=live2d_model,
                tts_preprocessor_config=tts_preprocessor_config,
                faster_first_response=basic_memory_settings.get(
                    "faster_first_response", True
                ),
                segment_method=basic_memory_settings.get(
                    "segment_method", "pysbd"
                ),
                use_mcpp=basic_memory_settings.get(
                    "use_mcpp", False
                ),
                interrupt_method=interrupt_method,
                tool_prompts=tool_prompts,
                tool_manager=tool_manager,
                tool_executor=tool_executor,
                mcp_prompt_string=mcp_prompt_string,
            )

        elif conversation_agent_choice == "mem0_agent":
            logger.critical(
                "mem0_agent is deprecated and no longer supported. Please switch to basic_memory_agent or a custom agent implementation."
            )
            raise NotImplementedError(
                "mem0_agent is deprecated and no longer supported."
            )

        elif conversation_agent_choice == "hume_ai_agent":
            settings = agent_settings.get(
                "hume_ai_agent", {}
            )
            return HumeAIAgent(
                api_key=settings.get("api_key"),
                host=settings.get("host", "api.hume.ai"),
                config_id=settings.get("config_id"),
                idle_timeout=settings.get(
                    "idle_timeout", 15
                ),
            )

        elif conversation_agent_choice == "letta_agent":
            settings = agent_settings.get("letta_agent", {})
            return LettaAgent(
                live2d_model=live2d_model,
                id=settings.get("id"),
                tts_preprocessor_config=tts_preprocessor_config,
                faster_first_response=settings.get(
                    "faster_first_response"
                ),
                segment_method=settings.get(
                    "segment_method"
                ),
                host=settings.get("host"),
                port=settings.get("port"),
            )

        # 在 else 分支中
        else:
            # ========== 动态导入自定义 Agent ==========
            try:
                module_path, class_name = (
                    conversation_agent_choice.rsplit(".", 1)
                )
                module = importlib.import_module(
                    module_path
                )
                agent_class = getattr(module, class_name)
                logger.info(
                    f"Dynamically loading custom agent: {agent_class} from {module_path}"
                )
            except (
                ImportError,
                AttributeError,
                ValueError,
            ) as e:
                logger.error(
                    f"Failed to load custom agent '{conversation_agent_choice}': {e}"
                )
                raise ValueError(
                    f"Unsupported agent type or failed to load custom agent: {conversation_agent_choice}"
                )

            # 🔧 将 agent_settings 转为字典（兼容 Pydantic 模型）
            if hasattr(agent_settings, "model_dump"):
                agent_settings_dict = (
                    agent_settings.model_dump()
                )
            else:
                agent_settings_dict = agent_settings

            custom_settings = agent_settings_dict.get(
                conversation_agent_choice, {}
            )
            logger.debug(
                f"Custom settings for {conversation_agent_choice}: {custom_settings}"
            )

            # 强制要求 llm_provider
            llm_provider = custom_settings.get(
                "llm_provider"
            )
            if not llm_provider:
                raise ValueError(
                    f"Custom agent '{conversation_agent_choice}' requires 'llm_provider' in its settings."
                )

            # 🔧 将 llm_configs 转为字典
            if hasattr(llm_configs, "model_dump"):
                llm_configs_dict = llm_configs.model_dump()
            else:
                llm_configs_dict = llm_configs

            llm_config = llm_configs_dict.get(llm_provider)
            if llm_config is None:
                raise ValueError(
                    f"Configuration not found for LLM provider: {llm_provider}"
                )

            # 弹出 interrupt_method（如果存在）
            interrupt_method = llm_config.pop(
                "interrupt_method", "user"
            )

            # 创建 LLM 实例
            llm = StatelessLLMFactory.create_llm(
                llm_provider=llm_provider,
                system_prompt=system_prompt,
                **llm_config,
            )

            # 构建基础参数
            base_args = {
                "llm": llm,
                "system": system_prompt,
                "live2d_model": live2d_model,
                "tts_preprocessor_config": tts_preprocessor_config,
                "tool_prompts": kwargs.get(
                    "system_config", {}
                ).get("tool_prompts", {}),
                "tool_manager": kwargs.get("tool_manager"),
                "tool_executor": kwargs.get(
                    "tool_executor"
                ),
                "mcp_prompt_string": kwargs.get(
                    "mcp_prompt_string", ""
                ),
                "interrupt_method": interrupt_method,
            }

            # 添加 BasicMemoryAgent 风格的额外参数
            for key in [
                "faster_first_response",
                "segment_method",
                "use_mcpp",
            ]:
                if key in custom_settings:
                    base_args[key] = custom_settings[key]

            # 剩余的自定义设置作为额外关键字参数
            reserved_keys = set(base_args.keys()) | {
                "llm_provider"
            }
            extra_kwargs = {
                k: v
                for k, v in custom_settings.items()
                if k not in reserved_keys
            }

            try:
                agent = agent_class(
                    **base_args, **extra_kwargs
                )
                logger.info(
                    f"Successfully instantiated custom agent: {conversation_agent_choice}"
                )
                return agent
            except Exception as e:
                logger.error(
                    f"Failed to instantiate custom agent '{conversation_agent_choice}': {e}"
                )
                raise
